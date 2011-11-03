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

replyRequest :: String -> String
replyRequest l = concat . suffixes $ l -- (concat . suffixes $ line) --concat . suffixes $ line

echoLoop hd = do
  input <- hGetLine hd
  let l = rstrip input
  putStrLn l
  let res = replyRequest l
  putStrLn res
  hPutStrLn hd res
  hFlush hd
  `catch` onError
  `finally` hClose hd

onError :: IOException -> IO ()
onError e = do
  return ()
