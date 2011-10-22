import qualified System.IO.UTF8 as U
import Text.Printf
data Page = Page  { pageName :: String  , pageSlug :: String  , pageContent :: String  } deriving Show

main = putStrLn $ pageSlug p where p = Page "suzuki" "hoge" "fuga"

