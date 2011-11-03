import Control.Concurrent
import Control.Concurrent.STM
import Control.Monad (unless, forever)

newtype Counter = Counter (TVar Int)

modifyTVar :: (a -> a) -> TVar a -> STM ()
modifyTVar f tv = readTVar tv >>= writeTVar tv . f

-- create a new counter and set its value 0.
newCounter :: IO Counter
newCounter = do tv <- newTVarIO 0
                return $ Counter tv

-- Increments the counter value.
incCounter :: Counter -> IO ()
incCounter (Counter c) = atomically $ modifyTVar (+1) c

-- Wait until the counter value becomes `n'
waitCounter :: Counter -> Int -> IO ()
waitCounter (Counter c) n = atomically $
                            do v <- readTVar c
                               unless (v == n) retry

-- forkIO with Counter.
fork :: Counter -> IO () -> IO ThreadId
fork counter act = forkIO $ act >> incCounter counter

-- threadDelay in milliseconds.
-- `delay 1000' means that this thread sleeps 1 sec at least.
delay :: Int -> IO ()
delay = threadDelay . (* 1000)

-- Do the first argument action every `time' ms infinitely.
every :: Int -> IO a -> IO ()
every time io = forever $ delay time >> io

main :: IO ()
main = do c <- newCounter
          fork c $ every 1000 $ putStrLn "a"
          fork c $ every 500 $ putStrLn "b"
          waitCounter c 2
