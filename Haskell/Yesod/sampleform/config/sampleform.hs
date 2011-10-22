{-# LANGUAGE CPP #-}
#if PRODUCTION
import Controller (withSampleForm)
import Network.Wai.Handler.Warp (run)

main :: IO ()
main = withSampleForm $ run 3000
#else
import Controller (withSampleForm)
import System.IO (hPutStrLn, stderr)
import Network.Wai.Middleware.Debug (debug)
import Network.Wai.Handler.Warp (run)

main :: IO ()
main = do
    let port = 3000
    hPutStrLn stderr $ "Application launched, listening on port " ++ show port
    withSampleForm $ run port . debug
#endif
