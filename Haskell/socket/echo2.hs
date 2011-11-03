module Main where
import Prelude hiding (catch)
import Network (listenOn, accept, sClose, Socket,
                withSocketsDo, PortID(..))
import System.IO
import System.Environment (getArgs)
import Control.Exception
import Control.Concurrent 
import Control.Concurrent.STM
import Control.Monad (forM, filterM, liftM, when)

main = withSocketsDo $ do 
    [portStr] <- getArgs
    let port = fromIntegral (read portStr :: Int)
    servSock <- listenOn $ PortNumber port
    putStrLn $ "listening on: " ++ show port
    start servSock `finally` sClose servSock 
start servSock = do
    acceptChan <- atomically newTChan
    forkIO $ acceptLoop servSock acceptChan
    mainLoop servSock acceptChan []
type Client = (TChan String, Handle)

acceptLoop :: Socket -> TChan Client -> IO ()
acceptLoop servSock chan = do
    (cHandle, host, port) <- accept servSock
    cChan <- atomically newTChan
    cTID  <- forkIO $ clientLoop cHandle cChan
    atomically $ writeTChan chan (cChan, cHandle)
    acceptLoop servSock chan


clientLoop :: Handle -> TChan String -> IO ()
clientLoop handle chan =
    listenLoop (hGetLine handle) chan
                   `catch` onClientError
                   `finally` hClose handle 

onClientError :: IOException -> IO ()
onClientError = const $ return ()

listenLoop :: IO a -> TChan a -> IO ()
listenLoop act chan = 
    sequence_ (repeat (act >>= atomically . writeTChan chan))

suffixes :: String -> [String]
suffixes []     = []
suffixes t@(x:xs) = t : suffixes xs

replyMessage :: String -> String
replyMessage l = ("res: " ++ l) ++ l -- (concat . suffixes $ line) --concat . suffixes $ line


mainLoop :: Socket -> TChan Client -> [Client] -> IO ()
mainLoop servSock acceptChan clients = do
    r <- atomically $ (Left `fmap` readTChan acceptChan)
                      `orElse`
                      (Right `fmap` tselect clients)
    case r of
          Left (ch,h)    -> do
               putStrLn "new client"
               mainLoop servSock acceptChan $ (ch,h):clients
          Right (line,_) -> do
               putStrLn $ "data: " ++ line
               clients' <- forM clients $ 
                            \(ch,h) -> do 
                                hPutStrLn h (replyMessage line)
                                hFlush h
                                return [(ch,h)] 
                            `catch` onMainLoopException h
               let dropped = length $ filter null clients'
               when (dropped > 0) $ 
                   putStrLn ("clients lost: " ++ show dropped)
               mainLoop servSock acceptChan $ concat clients'

onMainLoopException :: Handle -> IOException -> IO [(TChan String, Handle)]
onMainLoopException h e = const (hClose h >> return []) e

tselect :: [(TChan a, t)] -> STM (a, t)
tselect = foldl orElse retry
          . map (\(ch, ty) -> (flip (,) ty) `fmap` readTChan ch)
