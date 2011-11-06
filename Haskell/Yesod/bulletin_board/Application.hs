{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE CPP #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
module Application
    (  withDevelAppPort
    ) where

import Foundation
import Settings
import BulletinBoard

import Yesod
import Yesod.Static
import Yesod.Logger (makeLogger, flushLogger, Logger, logString, logLazyText)
import Control.Monad.IO.Class (liftIO)
import Control.Concurrent (forkIO, killThread)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)
import Network.Wai.Middleware.Debug (debugHandle)
import Data.Dynamic (Dynamic, toDyn)
import qualified System.Posix.Signals as Signal


mkYesodDispatch "BulletinBoard" resourcesBulletinBoard

withBulletinBoard :: AppConfig -> Logger -> (Application -> IO a) -> IO ()
withBulletinBoard conf logger f = do
    s <- static "static"
    Settings.withConnectionPool conf $ \p -> do
--        runConnectionPool (runMigration migrateAll) p
        let h = BulletinBoard s
        tid <- forkIO $ toWaiApp h >>= f >> return ()
        flag <- newEmptyMVar
        _ <- Signal.installHandler Signal.sigINT (Signal.CatchOnce $ do
            putStrLn "Caught an interrupt"
            killThread tid
            putMVar flag ()) Nothing
        takeMVar flag



-- for yesod devel
withDevelAppPort :: Dynamic
withDevelAppPort =
    toDyn go
  where
    go :: ((Int, Application) -> IO ()) -> IO ()
    go f = do
        conf <- Settings.loadConfig Settings.Development
        let port = 3000
        logger <- makeLogger
        logString logger $ "Devel application launched, listening on port " ++ show port
        withBulletinBoard conf logger $ \app -> f (port, debugHandle (logHandle logger) app)
        flushLogger logger
      where
        logHandle logger msg = logLazyText logger msg >> flushLogger logger

main :: IO ()
main = do
  s <- static "static"
  warpDebug 3000 $ BulletinBoard s

