{-# LANGUAGE CPP #-}
#if PRODUCTION
import Controller (withSuztomo)
import Network.Wai.Handler.Warp (run)

main :: IO ()
main = withSuztomo $ run 3000
#else
import Controller (withSuztomo)
import System.IO (hPutStrLn, stderr)
import Network.Wai.Middleware.Debug (debug)
import Network.Wai.Handler.Warp (run)

main :: IO ()
main = do
    let port = 3000
    hPutStrLn stderr $ "Application launched, listening on port " ++ show port
    withSuztomo $ run port . debug
#endif
