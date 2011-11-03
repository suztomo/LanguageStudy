{-# LANGUAGE QuasiQuotes, TemplateHaskell, MultiParamTypeClasses,
    OverloadedStrings, TypeFamilies, GADTs #-}
import Yesod
import Yesod.Static

import Yesod.Form.Jquery
import Yesod.Form
import Data.Time
import qualified Data.Text as T
import Control.Applicative ((<$>), (<*>))
import Data.Text
import Database.Persist
import Database.Persist.Sqlite
import Database.Persist.TH
import Control.Monad.IO.Class (liftIO)
import Maybe
import qualified Data.ByteString.Char8 as C8
import qualified Data.Digest.Pure.SHA as SHA
import qualified Data.ByteString.Lazy as BL
import qualified Codec.Binary.UTF8.String as U8
import Text.Julius

data BulletinBoard = BulletinBoard {
      bbStatic :: Static
    }

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

mkYesod "BulletinBoard" [parseRoutes|
/ RootR GET
/user/add_form AddUserFormR GET
/user/add AddUserR POST
/user/auth_form AuthUserFormR GET
/user/auth AuthUserR POST
/user/info LoginUserInfoR GET
/add AddPostR GET
/logout LogoutR GET
/static StaticR Static bbStatic
|]

instance Yesod BulletinBoard where
    approot _ = ""
    defaultLayout widget = do
      content <- widgetToPageContent widget
      hamletToRepHtml [hamlet|
\<!DOCTYPE html>

<html>
  <head>
    <title>#{pageTitle content}
    <link rel="stylesheet" href="@{StaticR css_screen_css}">
    <script src="/static/jquery-1.6.4.min.js">
    <script src="@{StaticR script_js}">
    \^{pageHead content}
  <body>
    <p id="user_info">
    <div id="content">
      \^{pageBody content}
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


postForm :: Maybe Post -> Html -> Form BulletinBoard BulletinBoard (FormResult Post, Widget)
postForm p = do
  renderDivs $ Post
           <$> areq textareaField "text" Nothing

userForm :: Html -> Form BulletinBoard BulletinBoard (FormResult User, Widget)
userForm = renderDivs $ User
           <$> areq textField "Name" Nothing
           <*> areq textField "User ID" Nothing
           <*> areq passwordField "Password" Nothing

showUserForm widget enctype = defaultLayout [whamlet|
<h1>User Add
<form method=post action=@{AddUserR} enctype=#{enctype}>
    ^{widget}
    <input type=submit
|]

showUserAuthForm widget enctype = defaultLayout [whamlet|
<h1>Authentication
<form method=post action=@{AuthUserR} enctype=#{enctype}>
    ^{widget}
    <input type=submit
<a href=@{AddUserFormR}>
    Register
|]

userAuthForm :: Html -> Form BulletinBoard BulletinBoard (FormResult UserAuth,
                                                                     Widget)
userAuthForm = renderDivs $ UserAuth
               <$> areq textField "User ID" Nothing
               <*> areq passwordField "Password" Nothing

getLogoutR :: Handler RepHtml
getLogoutR = do
  deleteSession "user"
  deleteSession "name"
  redirectText RedirectSeeOther (T.pack "/")

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
                 redirectText RedirectSeeOther (T.pack "/")
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
                                           redirectText RedirectSeeOther (T.pack "/")
                    _ -> defaultLayout [whamlet| Invalid password |]
    _ -> showUserForm widget enctype

getLoginUserInfoR :: Handler RepHtmlJson
getLoginUserInfoR = do
  defaultUserName <- lookupSession "user"
  case defaultUserName of
    Nothing -> redirectText RedirectSeeOther (T.pack "/user/auth_form")
    Just userId -> do
      mKeyUser <- liftIO $ fetchUser userId
      case mKeyUser of
        Nothing -> do
          redirectText RedirectSeeOther (T.pack "/user/auth_form#nosuchuser")
        Just (_, bbuser) -> defaultLayoutJson  (return ()) (json bbuser)
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
    Nothing -> redirectText RedirectSeeOther (T.pack "/user/auth_form")
    Just name -> do
             bbPostKeyList <- liftIO $ getPostList
             let postList = Prelude.map (\(x,y) -> y) bbPostKeyList :: [ BBPost ]
             ((_, widget), enctype) <- generateFormGet . postForm $ Just (Post (Textarea ""))
             defaultLayout $ do
                         addJulius $(juliusFile "foo.julius")
                         [whamlet|
<h1>なにしてるんですか
<form method=get action=@{AddPostR} enctype=#{enctype}>
    ^{widget}
    <input type=submit>

<ul>
$forall post <- postList
    <li>#{bBPostText post}
         <span style="font-size:0.7em">
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
                redirectText RedirectSeeOther (T.pack "/user/auth_form")
    Just authorId -> do
        mKeyUser <- liftIO $ fetchUser authorId
        case mKeyUser of
          Nothing -> do
                liftIO $ putStrLn "lookupUser failed"
                redirectText RedirectSeeOther (T.pack "/user/auth_form")
          Just (key, author) -> do
                ((result, widget), enctype) <- runFormGet $ postForm Nothing
                case result of
                  FormSuccess post -> do
                              liftIO $ savePost post author key
                              liftIO $ putStrLn (show post)
                              redirectText RedirectSeeOther (T.pack "/")
                  _ -> sendResponse $ RepPlain $ toContent("Form failed. but thank you for accessing this" :: String)

--  sendResponse $ RepPlain $ toContent("Thank you for accessing this" :: String)

main :: IO ()
main = do
  s <- static "static"
  warpDebug 3000 $ BulletinBoard s