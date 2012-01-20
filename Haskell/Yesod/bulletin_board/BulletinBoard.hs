{-# LANGUAGE QuasiQuotes, TemplateHaskell, MultiParamTypeClasses,
    OverloadedStrings, TypeFamilies, GADTs, FlexibleContexts #-}
module BulletinBoard where

import Yesod
import Yesod.Static

import Settings
import Foundation

import Yesod.Form.Jquery
--import Yesod.Form
--import MForm
import Yesod.Logger (makeLogger, flushLogger, Logger, logString, logLazyText)
import Data.Time
import qualified Data.Text as T
import Control.Applicative ((<$>), (<*>))
import Data.Text
import Data.Dynamic (Dynamic, toDyn)
import Database.Persist
import Database.Persist.Sqlite
import Database.Persist.TH
import Control.Monad.IO.Class (liftIO)
import Control.Concurrent (forkIO, killThread)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)
import qualified Data.ByteString.Char8 as C8
import qualified Data.Digest.Pure.SHA as SHA
import qualified Data.ByteString.Lazy as BL
import qualified Codec.Binary.UTF8.String as U8
import Network.Wai.Middleware.RequestLogger (logHandle)


-- import Text.Julius


staticFiles "static"

share [mkPersist sqlSettings, mkMigrate "migrateAll"] [persist|
BBPost
    name String Maybe
    text String
    authorId BBUserId
    created UTCTime default=now

BBUser
    name String
    userId String
    password String
    created UTCTime default=now
|]

syncdb :: IO ()
syncdb = withSqliteConn "db.sqlite" $ runSqlConn $ do
           runMigration migrateAll
           return ()

-- mkYesod "BulletinBoard" [parseRoutes|
-- / UserR GET
-- /home RootR GET
-- /user/add_form AddUserFormR GET
-- /user/add AddUserR POST
-- /user/auth_form AuthUserFormR GET
-- /user/auth AuthUserR POST
-- /user/info LoginUserInfoR GET
-- /add AddPostR GET
-- /logout LogoutR GET
-- /static StaticR Static bbStatic
-- |]



instance Yesod BulletinBoard where
    approot _ = ""
    defaultLayout widget = do
      content <- widgetToPageContent widget
      hamletToRepHtml [hamlet|
\<!DOCTYPE html>

<html>
  <head>
    <title>#{pageTitle content}
    <link rel="stylesheet" href="@{StaticR bootstrap_min_css}"
    <link rel="stylesheet" href="@{StaticR style_css}"
    <script src="/static/jquery-1.6.4.min.js"
    <script src="@{StaticR script_js}"
    \^{pageHead content}
  <body>
    <div class="topbar">
      <div class="fill">
        <div class="container">
          <a class="brand" href="#">Wassr</a>
          <ul class="nav">
            <li class="active"><a href="/home">Home</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#contact">Contact</a></li>
          </ul>
          <p id="user_info" # class="pull-right">

    <div class="container">
      <div class="content">
        \^{pageBody content}
      <footer>
        <p>&copy; Suztomo 2011

|]

instance RenderMessage BulletinBoard FormMessage where
    renderMessage _ _ = defaultFormMessage

data Post = Post {
      text :: Textarea
} deriving Show

data User = User {
      userName :: Text,
      userId :: Text,
      passwordUnHashed :: Text
    } deriving Show

data UserAuth = UserAuth {
      authUserId :: Text,
      authPasswordUnHashed :: Text
    } deriving Show


--postForm :: Maybe Post -> Html -> Form BulletinBoard BulletinBoard (FormResult Post, Widget)
--postForm p = do
--  renderDivs $ Post <$> areq textareaField "" Nothing

postForm :: Maybe Post -> Html -> MForm BulletinBoard BulletinBoard (FormResult Post, Widget)
postForm p extra = do
  (textRes, textView) <- mreq textareaField "" Nothing
  let rs = Post <$> textRes
  let w = do
        toWidget [lucius|
##{fvId textView} {

}
|]
        [whamlet|
#{extra}
<p class="post-text-area"
    ^{fvInput textView}
|]
  return (rs, w)

  

userForm :: Html -> MForm BulletinBoard BulletinBoard (FormResult User, Widget)
userForm = renderDivs $ User
           <$> areq textField "Name" Nothing
           <*> areq textField "User ID" Nothing
           <*> areq passwordField "Password" Nothing

showUserForm widget enctype = defaultLayout [whamlet|
<h1>User Add
<form method=post action=@{AddUserR} enctype=#{enctype}>
    ^{widget}
    <input type=submit  class="btn primary"
|]

showUserAuthForm widget enctype = defaultLayout [whamlet|
<h1>Authentication
<form method=post action=@{AuthUserR} enctype=#{enctype}>
    ^{widget}
    <input type=submit  class="btn primary"
<a href=@{AddUserFormR}>
    Register
|]

userAuthForm :: Html -> MForm BulletinBoard BulletinBoard (FormResult UserAuth,
                                                                     Widget)
userAuthForm = renderDivs $ UserAuth
               <$> areq textField "User ID" Nothing
               <*> areq passwordField "Password" Nothing

getUserR :: Handler RepHtml
getUserR = do
  defaultLayout $ do
             [whamlet| user info? |]

getLogoutR :: Handler RepHtml
getLogoutR = do
  deleteSession "user"
  deleteSession "name"
  redirect RedirectSeeOther RootR

getAddUserFormR :: Handler RepHtml
getAddUserFormR = do
  ((_, widget), enctype) <- generateFormPost userForm
  showUserForm widget enctype

postAddUserR :: Handler RepHtml
postAddUserR = do
  ((result, widget), enctype) <- runFormPost userForm
  case result of
    FormSuccess user -> do
                 bbuser <- liftIO $ saveUser user
                 liftIO $ putStrLn (show bbuser)
                 setSession "user" $ pack (bBUserUserId bbuser)
                 setSession "name" $ pack (bBUserName bbuser)
                 redirect RedirectSeeOther RootR
    _ -> showUserForm widget enctype

getAuthUserFormR :: Handler RepHtml
getAuthUserFormR = do
  ((_, widget), enctype) <- generateFormPost userAuthForm
  showUserAuthForm widget enctype

fetchUser :: Text -> IO (Maybe (BBUserId, BBUser))
fetchUser userIdText = withSqliteConn "db.sqlite" $ runSqlConn $ do
    let 
        userId = T.unpack userIdText
    oneUser <- (selectList [ BBUserUserId ==. userId ] [])
--    liftIO $ print (oneUser :: [(BBUserId, BBUser)])
    case oneUser of
      (k, u):_ -> return (Just (k,u))
      _          -> return Nothing


postAuthUserR :: Handler RepHtml
postAuthUserR = do
  ((result, widget), enctype) <- runFormPost userAuthForm
  case result of
    FormSuccess userAuth -> do
                  mbbuser <- liftIO $ fetchUser (authUserId userAuth)
                  case mbbuser of
                    Just (_, bbuser) | (bBUserPassword bbuser) == textOfSHA1 (authPasswordUnHashed userAuth) -> do
                                           setSession "user" $ pack (bBUserUserId bbuser)
                                           setSession "name" $ pack (bBUserName bbuser)
                                           redirect RedirectSeeOther RootR

                    _ -> defaultLayout [whamlet| Invalid password |]
    _ -> showUserForm widget enctype

getLoginUserInfoR :: Handler RepJson
getLoginUserInfoR = do
  defaultUserName <- lookupSession "user"
  case defaultUserName of
    Nothing -> redirect RedirectSeeOther RootR
    Just userId -> do
      mKeyUser <- liftIO $ fetchUser userId
      case mKeyUser of
        Nothing -> redirect RedirectSeeOther AuthUserFormR
        Just (_, bbuser) -> jsonToRepJson $ json bbuser
          where
            json bbuser = jsonMap
                          [ ("name", jsonScalar $ bBUserName bbuser),
                            ("login", jsonScalar $ bBUserUserId bbuser),
                            ("id", jsonScalar $ bBUserUserId bbuser) ]

getRootR :: Handler RepHtml
getRootR = do
  session <- getSession
  defaultUserName <- lookupSession "user"
  case defaultUserName of
    Nothing -> redirect RedirectSeeOther AuthUserFormR
    Just name -> do
             bbPostKeyList <- liftIO $ getPostList
             let postList = Prelude.map (\(x,y) -> y) bbPostKeyList :: [ BBPost ]
             ((_, widget), enctype) <- generateFormGet . postForm $ Just (Post (Textarea ""))
             defaultLayout $ do
                         addJulius $(juliusFile "foo")
                         [whamlet|
<h1>なにしてるんですか
<form method=get action=@{AddPostR} enctype=#{enctype}>
    <div class="pull-left"
        ^{widget}
    <input type=submit  class="btn primary textsubmit"

<ul class="post-list unstyled">
    $forall post <- postList
        <li>#{bBPostText post}
            <span class="author-name"
                $maybe name <- bBPostName post
                    by #{ name }
                $nothing
                    No Name
|]
--               #{showName $ bBPostName post}
getPostList :: IO [ (BBPostId, BBPost) ]
getPostList = withSqliteConn "db.sqlite" $ runSqlConn $ do
    selectList [] [ Desc BBPostId ]

--newtype BBUserKey = Key SqlPersist BBUser

savePost :: Post -> BBUser -> BBUserId -> IO ()
savePost post author authorKey = withSqliteConn "db.sqlite" $ runSqlConn $ do
    time <- liftIO $ getCurrentTime
    let postName = Just (bBUserName author)
        postText = T.unpack $ unTextarea $ text post
    postId <- insert $ BBPost postName postText authorKey time
    return ()

textOfSHA1 :: Text -> String
textOfSHA1 = SHA.showDigest . SHA.sha1 . BL.pack . U8.encode . T.unpack

saveUser :: User -> IO BBUser
saveUser user = withSqliteConn "db.sqlite" $ runSqlConn $ do
  time <- liftIO $ getCurrentTime
  let name = T.unpack $ userName user
      a = T.unpack $ userId user :: String
      password = textOfSHA1 $ passwordUnHashed user
  let bbuser = BBUser name a password time
  insert $ bbuser
  return bbuser


getAddPostR :: Handler RepHtml
getAddPostR = do
  defaultUserName <- lookupSession "user"
  case defaultUserName of
    Nothing -> do
                liftIO $ putStrLn "lookupSession failed"
                redirect RedirectSeeOther AuthUserFormR
    Just authorId -> do
        mKeyUser <- liftIO $ fetchUser authorId
        case mKeyUser of
          Nothing -> do
                liftIO $ putStrLn "lookupUser failed"
                redirect RedirectSeeOther AuthUserFormR
          Just (key, author) -> do
                ((result, widget), enctype) <- runFormGet $ postForm Nothing
                case result of
                  FormSuccess post -> do
                              liftIO $ savePost post author key
                              liftIO $ putStrLn (show post)
                              redirect RedirectSeeOther RootR
                  _ -> sendResponse $ RepPlain $ toContent("Form failed. but thank you for accessing this" :: String)


