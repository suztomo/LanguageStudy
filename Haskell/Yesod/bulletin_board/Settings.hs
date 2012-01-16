{-# LANGUAGE CPP #-}
{-# LANGUAGE QuasiQuotes, TemplateHaskell, MultiParamTypeClasses,
    OverloadedStrings, TypeFamilies, GADTs, FlexibleContexts, CPP #-}

-- | Settings are centralized, as much as possible, into this file. This
-- includes database connection settings, static file locations, etc.
-- In addition, you can configure a number of different aspects of Yesod
-- by overriding methods in the Yesod typeclass. That instance is
-- declared in the Foundation.hs file.
module Settings
    ( hamletFile
    , cassiusFile
    , juliusFile
    , luciusFile
    , widgetFile
    , PersistConfig
    , staticRoot
    , staticDir
    ) where


import Text.Shakespeare.Text (st)
import Language.Haskell.TH.Syntax
import Database.Persist.Sqlite (SqliteConf)
import Yesod.Default.Config
import qualified Yesod.Default.Util
import Data.Text (Text)

import qualified Text.Hamlet as H
import qualified Text.Cassius as H
import qualified Text.Julius as H
import qualified Text.Lucius as H
import Language.Haskell.TH.Syntax
import Data.Monoid (mempty, mappend)

-- | Which Persistent backend this site is using.
type PersistConfig = SqliteConf

-- Static setting below. Changing these requires a recompile

-- | The location of static files on your system. This is a file system
-- path. The default value works properly with your scaffolded site.
staticDir :: FilePath
staticDir = "static"

-- | The base URL for your static files. As you can see by the default
-- value, this can simply be "static" appended to your application root.
-- A powerful optimization can be serving static files from a separate
-- domain name. This allows you to use a web server optimized for static
-- files, more easily set expires and cache values, and avoid possibly
-- costly transference of cookies on static files. For more information,
-- please see:
--   http://code.google.com/speed/page-speed/docs/request.html#ServeFromCookielessDomain
--
-- If you change the resource pattern for StaticR in Foundation.hs, you will
-- have to make a corresponding change here.
--
-- To see how this value is used, see urlRenderOverride in Foundation.hs
staticRoot :: AppConfig DefaultEnv x ->  Text
staticRoot conf = [st|#{appRoot conf}/static|]


-- The rest of this file contains settings which rarely need changing by a
-- user.

widgetFile :: String -> Q Exp
#if DEVELOPMENT
widgetFile = Yesod.Default.Util.widgetFileReload
#else
widgetFile = Yesod.Default.Util.widgetFileNoReload
#endif

globFile :: String -> String -> FilePath
globFile kind x = kind++"/"++x ++"." ++ kind

hamletFile :: FilePath -> Q Exp
hamletFile = H.hamletFile . globFile "hamlet"


cassiusFile :: FilePath -> Q Exp
cassiusFile = 
#ifdef PRODUCTION
  H.cassiusFile . globFile "cassius"
#else
  H.cassiusFileDebug . globFile "cassius"
#endif

luciusFile :: FilePath -> Q Exp
luciusFile = 
#ifdef PRODUCTION
  H.luciusFile . globFile "lucius"
#else
  H.luciusFileDebug . globFile "lucius"
#endif

juliusFile :: FilePath -> Q Exp
juliusFile =
#ifdef PRODUCTION
  H.juliusFile . globFile "julius"
#else
  H.juliusFileDebug . globFile "julius"
#endif
