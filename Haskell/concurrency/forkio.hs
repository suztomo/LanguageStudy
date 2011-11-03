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
  aid <- forkOS $ threadA 31
  bid <- forkOS $ threadA 31
  cid <- forkOS $ threadA 31
  threadA 31

threadA :: Int -> IO ()
threadA x = do
  let k = tarai x 22 11
  putStrLn (show k)
