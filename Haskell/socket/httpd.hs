import Network (listenOn, accept, sClose, Socket,
                 withSocketsDo, PortID(..))
import Prelude hiding (catch)
import System
import System.IO
import System.Environment (getArgs)
import Control.Exception
import Control.Concurrent
import Control.Concurrent.STM
import Control.Monad (forM, filterM, liftM, when)
import Data.Char (isSpace)

rstrip = reverse . dropWhile isSpace . reverse

main :: IO ()
main = withSocketsDo $ do
         let port = 8000
         soc <- listenOn $ PortNumber port
         putStrLn $ "start server, listening on: " ++ show port
         acceptLoop soc `finally` sClose soc

acceptLoop :: Socket -> IO ()
acceptLoop soc = do
  (hd, host, port) <- accept soc
--  forkOS $echoLoop hd
  echoLoop hd
  acceptLoop soc


suffixes :: String -> [String]
suffixes []     = []
suffixes t@(x:xs) = t : suffixes xs

replyRequest :: String -> IO String
replyRequest = return . concat . suffixes -- (concat . suffixes $ line) --concat . suffixes $ line

reply :: [String] -> IO String
reply l = do
  putStrLn $ concat l
  return "HTTP/1.1 200 OK\nDate: Thu, 03 Nov 2011 06:24:54 GMT\n\nHello, HTTP Server"

readHeaders :: Handle -> IO [String]
readHeaders hd = readUntilTwoNewLines hd
    where readUntilTwoNewLines hd = do
            input <- hGetLine hd
            if ("\r" == input)
               then return []
               else do
                 rest <- readUntilTwoNewLines hd
                 return ((rstrip input) : rest)

onLastLine :: IOError -> IO [a]
onLastLine e = return []

echoLoop :: Handle -> IO ()
echoLoop hd = do
  lines <- readHeaders hd
  res <- reply lines
  hPutStrLn hd $ res ++ (concat lines)
  hFlush hd
  `catch` onError
  `finally` hClose hd

onError :: IOException -> IO ()
onError e = do
  return ()
