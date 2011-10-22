{-# LANGUAGE TypeFamilies, QuasiQuotes, TemplateHaskell, MultiParamTypeClasses, OverloadedStrings #-}
import Yesod
import Web.Routes.Quasi.Parse
import qualified Text.Blaze.Html5 as H

data Test = Test {
  }

mkYesod "Test" [parseRoutesNoCheck|
/page/#String      PageR   GET
/user/#String      UserR   GET
|]

instance Yesod Test where
  approot _ = "" 
  defaultLayout widget = do
                      content <- widgetToPageContent widget
                      hamletToRepHtml [hamlet|
\<!DOCTYPE html>

<html>
  <head>
    <title>#{pageTitle content}
  <body>
    <ul id="navbar">
    <div id="content">
      \^{pageBody content}
|]


getUserR :: String -> Handler RepHtml
getUserR userName = defaultLayout
                    (do
                      setTitle $ H.toHtml $ "Hello " ++ userName
                      addHamlet $ html userName
                    )
    where
      html page = [hamlet|
 <h1>User: #{userName}
 <p>This page is for user: #{userName}
 |]

getPageR :: String -> Handler RepHtml
getPageR pageName = defaultLayout
                    (do
                      setTitle $ H.toHtml $ "Article: " ++ pageName
                      addHamlet $ html pageName
                    )
    where
      html page = [hamlet|
 <h1>Page: #{pageName}
 <p>This page is for page: #{pageName}
 |]

main :: IO ()
main = do
  warpDebug 3000 $ Test
