{-# LANGUAGE TypeFamilies, QuasiQuotes #-}
{-# LANGUAGE TypeFamilies, QuasiQuotes, MultiParamTypeClasses, TemplateHaskell, OverloadedStrings #-}
import Yesod
data HelloWorld = HelloWorld
mkYesod "HelloWorld" [$parseRoutes|
/user/#String/#Int NameR GET
/ HomeR GET
|]
instance Yesod HelloWorld where approot _ = ""
getHomeR :: GHandler sub HelloWorld RepHtml
getHomeR = defaultLayout [$hamlet|こんにちわ、世界!|]
getNameR :: String -> Int -> GHandler sub HelloWorld RepHtml
getNameR name age = defaultLayout [$hamlet|こんにちわ、#{name} (#{age})さん|]

main :: IO ()                                  
main = warpDebug 3000 HelloWorld
