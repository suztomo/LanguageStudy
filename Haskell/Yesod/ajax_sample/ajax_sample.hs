{-# LANGUAGE TypeFamilies, QuasiQuotes, TemplateHaskell, MultiParamTypeClasses, OverloadedStrings #-}
import Yesod
import Yesod.Static
import Data.Monoid (mempty)
import qualified Text.Blaze.Html5 as H
import Web.Routes.Quasi.Parse

data Page = Page  {
      pageName :: String,
      pageSlug :: String,
      pageContent :: String
    }

loadPages :: IO [Page]
loadPages = return
  [ Page "Page 1" "page-1" "My first page"
  , Page "Page 2" "page-2" "My second page"
  , Page "Page 3" "page-3" "My third page"
  ]

data Ajax = Ajax {
      ajaxPages :: [Page],
      ajaxStatic :: Static
  }

staticFiles "static/yesod/ajax"

mkYesod "Ajax" [parseRoutesNoCheck|
/                  HomeR   GET
/page/#String      PageR   GET
/static            StaticR Static ajaxStatic
/user/#String           UserR   GET
|]

instance Yesod Ajax where
  approot _ = "" 
  defaultLayout widget = do
                      Ajax pages _ <- getYesod
                      content <- widgetToPageContent widget
                      hamletToRepHtml [hamlet|
\<!DOCTYPE html>

<html>
  <head>
    <title>#{pageTitle content}
    <link rel="stylesheet" href="@{StaticR style_css}">
    <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.4.2/jquery.min.js">
    <script src="@{StaticR script_js}">
    \^{pageHead content}
  <body>
    <ul id="navbar">
      $forall page <- pages
        <li>
          <a href="@{PageR (pageSlug page)}">#{pageName page}
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
 <h1>#{userName}
 <p>This page is for #{userName}
 |]

getHomeR :: Handler ()
getHomeR = do
  Ajax pages _ <- getYesod
  let first = head pages
  redirect RedirectTemporary $ PageR $ pageSlug first


getPageR :: String -> Handler RepHtmlJson
getPageR slug = do
  Ajax pages _ <- getYesod
  case filter (\e -> pageSlug e == slug) pages of
      [] -> notFound
      page:_ -> defaultLayoutJson (do
          setTitle $ H.toHtml $ pageName page
          addHamlet $ html page
          ) (json page)
 where
  html page = [hamlet|
<h1>#{pageName page}
<article>#{pageContent page}
|]
  json page = jsonMap
      [ ("name", jsonScalar $ pageName page)
      , ("content", jsonScalar $ pageContent page)
      ]

main :: IO ()
main = do
  pages <- loadPages
  s <- static "static/yesod/ajax"
  warpDebug 3000 $ Ajax pages s
