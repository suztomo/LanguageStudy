{-# LANGUAGE PackageImports #-}
import "testproj" Controller (withDevelApp)
import Data.Dynamic (fromDynamic)
import Network.Wai.Handler.Warp (run)
import Network.Wai.Middleware.Debug (debug)
import Data.Maybe (fromJust)
import Control.Concurrent (forkIO)
import System.Directory (doesFileExist, removeFile)
import Control.Concurrent (threadDelay)

main :: IO ()
main = do
    putStrLn "Starting app"
    forkIO $ (fromJust $ fromDynamic withDevelApp) $ run 3000
    loop

loop :: IO ()
loop = do
    threadDelay 100000
    e <- doesFileExist "dist/devel-flag"
    if e then removeFile "dist/devel-flag" else loop
