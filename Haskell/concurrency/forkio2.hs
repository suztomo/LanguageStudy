import Control.Concurrent
import Control.Monad (unless, forever)

tarai :: Int -> Int -> Int -> Int
tarai x y z
    | x <= y    = z
    | otherwise = tarai(tarai (x-1) y z)
                       (tarai (y-1) z x)
                       (tarai (z-1) x y)
main :: IO ()
main = do
  aid <- forkOS $ threadA
  bid <- forkOS $ threadA
  cid <- forkOS $ threadA
  threadA

delay :: Int -> IO ()
delay = threadDelay . (* 1000)

every :: Int -> IO a -> IO ()
every time io = forever $ delay time >> io

threadA :: IO ()
threadA = do
  let k = tarai 33 22 11
  putStrLn (show k)
