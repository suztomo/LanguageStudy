module Paths_testproj (
    version,
    getBinDir, getLibDir, getDataDir, getLibexecDir,
    getDataFileName
  ) where

import Data.Version (Version(..))
import System.Environment (getEnv)

version :: Version
version = Version {versionBranch = [0,0,0], versionTags = []}

bindir, libdir, datadir, libexecdir :: FilePath

bindir     = "/Users/suztomo/.cabal/bin"
libdir     = "/Users/suztomo/.cabal/lib/testproj-0.0.0/ghc-6.12.1"
datadir    = "/Users/suztomo/.cabal/share/testproj-0.0.0"
libexecdir = "/Users/suztomo/.cabal/libexec"

getBinDir, getLibDir, getDataDir, getLibexecDir :: IO FilePath
getBinDir = catch (getEnv "testproj_bindir") (\_ -> return bindir)
getLibDir = catch (getEnv "testproj_libdir") (\_ -> return libdir)
getDataDir = catch (getEnv "testproj_datadir") (\_ -> return datadir)
getLibexecDir = catch (getEnv "testproj_libexecdir") (\_ -> return libexecdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "/" ++ name)
