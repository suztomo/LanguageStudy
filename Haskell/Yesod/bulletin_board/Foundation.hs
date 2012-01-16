{-# LANGUAGE QuasiQuotes, TemplateHaskell, TypeFamilies #-}
{-# LANGUAGE OverloadedStrings, MultiParamTypeClasses #-}
{-# LANGUAGE CPP #-}

module Foundation
    ( BulletinBoard (..)
    , BulletinBoardRoute (..)
    , resourcesBulletinBoard
    , Handler
    , Widget
    ) where


import Settings
import Yesod hiding (AppConfig, withYamlEnvironment, appEnv)

import Yesod.Default.Config
import Yesod.Default.Main

import Yesod.Static (Static, base64md5, StaticRoute(..))
import Yesod.Logger (Logger, logLazyText)
import System.Directory
import qualified Data.ByteString.Lazy as L
import Database.Persist.GenericSql
import Database.Persist.Base
import Data.Maybe (isJust)
import Control.Monad (join, unless)
import Network.Mail.Mime
import qualified Data.Text.Lazy.Encoding
import qualified Data.Text as T
import Web.ClientSession (getKey)
import Text.Blaze.Renderer.Utf8 (renderHtml)
import Text.Hamlet (shamlet)
import Text.Shakespeare.Text (stext)

data BulletinBoard = BulletinBoard {
      conf :: AppConfig DefaultEnv (),
      logger :: Logger,
      bbStatic :: Static,
      pool :: PersistConfigPool Settings.PersistConfig
    }

mkYesodData "BulletinBoard" [parseRoutes|
/ UserR GET
/home RootR GET
/user/add_form AddUserFormR GET
/user/add AddUserR POST
/user/auth_form AuthUserFormR GET
/user/auth AuthUserR POST
/user/info LoginUserInfoR GET
/add AddPostR GET
/logout LogoutR GET
/static StaticR Static bbStatic
|]
