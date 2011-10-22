{-# LANGUAGE QuasiQuotes, TemplateHaskell, MultiParamTypeClasses,
    OverloadedStrings, TypeFamilies, GADTs #-}
import Yesod
import Yesod.Form.Jquery
import Yesod.Form
import Data.Time
import Data.Text
import Control.Applicative ((<$>), (<*>))
import Data.Text
import Database.Persist
import Database.Persist.Sqlite
import Database.Persist.TH
import Control.Monad.IO.Class (liftIO)
import Maybe

data BulletinBoard = BulletinBoard

share [mkPersist sqlSettings, mkMigrate "migrateAll"] [persist|
BBPost
    name String Maybe
    text String
    created UTCTime default=now
|]

syncdb :: IO ()
syncdb = withSqliteConn "db.sqlite" $ runSqlConn $ do
           runMigration migrateAll
           return ()

mkYesod "BulletinBoard" [parseRoutes|
/ RootR GET
/add AddPostR GET
|]

instance Yesod BulletinBoard where
    approot _ = ""

instance RenderMessage BulletinBoard FormMessage where
    renderMessage _ _ = defaultFormMessage

data Post = Post {
      text :: Textarea,
      name :: Maybe Text
    } deriving Show


postForm :: Html -> Form BulletinBoard BulletinBoard (FormResult Post, Widget)
postForm = renderDivs $ Post
           <$> areq textareaField "text" Nothing
           <*> aopt textField "Name" Nothing

showName :: Maybe String -> String
showName ms = case ms of
                Just a -> a
                Nothing -> "No name"

getRootR :: Handler RepHtml
getRootR = do
  bbPostKeyList <- liftIO $ getPostList
  let postList = Prelude.map (\(x,y) -> y) bbPostKeyList :: [ BBPost ]
  ((_, widget), enctype) <- generateFormGet postForm
  defaultLayout [whamlet|
<h1>なにしてるんですか
<form method=get action=@{AddPostR} enctype=#{enctype}>
    ^{widget}
    <input type=submit>

<ul>
$forall post <- postList
    <li>#{bBPostText post}
         <span style="font-size:0.7em">
               #{showName $ bBPostName post}
|]

getPostList :: IO [ (BBPostId, BBPost) ]
getPostList = withSqliteConn "db.sqlite" $ runSqlConn $ do
    selectList [] [ Desc BBPostId ]

savePost :: Post -> IO ()
savePost post = withSqliteConn "db.sqlite" $ runSqlConn $ do
    time <- liftIO $ getCurrentTime
    let postName = fmap unpack (name post)
        postText = unpack $ unTextarea $ text post
    postId <- insert $ BBPost postName postText time
    return ()

getAddPostR :: Handler RepHtml
getAddPostR = do
    ((result, widget), enctype) <- runFormGet postForm
    case result of
        FormSuccess post -> do
                  liftIO $ savePost post
                  defaultLayout [whamlet|
<p>Saved! : #{show post}
<a href=@{RootR}>Back
|]
        _ -> sendResponse $ RepPlain $ toContent("Form failed. but thank you for accessing this" :: String)

--  sendResponse $ RepPlain $ toContent("Thank you for accessing this" :: String)

main :: IO ()
main = warpDebug 3000 BulletinBoard