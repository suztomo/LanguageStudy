{-# LANGUAGE CPP #-}
#if PRODUCTION
import Controller (withSuzmeg)
import Network.Wai.Handler.Warp (run)

main :: IO ()
main = withSuzmeg $ run 3000
#else
import Controller (withSuzmeg)
import System.IO (hPutStrLn, stderr)
import Network.Wai.Middleware.Debug (debug)
import Network.Wai.Handler.Warp (run)

main :: IO ()
main = do
    let port = 3000
    hPutStrLn stderr $ "Application launched, listening on port " ++ show port
    withSuzmeg $ run port . debug
#endif
