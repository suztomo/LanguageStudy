import Yesod.Default.Config (fromArgs)
import Yesod.Default.Main   (defaultMain)
import Application          (withBbbb)
import Prelude              (IO)

main :: IO ()
main = defaultMain fromArgs withBbbb