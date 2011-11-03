import Control.Concurrent
import Control.Monad (forever)

-- threadDelay in milliseconds.
-- `delay 1000' means that this thread sleeps 1 sec at least.
delay :: Int -> IO ()
delay = threadDelay . (* 1000)

-- Do the first argument action every `time' ms infinitely.
every :: Int -> IO a -> IO ()
every time io = forever $ delay time >> io

main :: IO ()
main = do forkIO $ every 1000 $ putStrLn "a"
          forkIO $ every 500 $ putStrLn "b"
          return ()
