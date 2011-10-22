{-# LANGUAGE QuasiQuotes, TemplateHaskell, TypeFamilies, OverloadedStrings, PackageImports #-}
{-# LANGUAGE GADTs #-}
import Database.Persist
import Database.Persist.Sqlite
import Database.Persist.TH
import Control.Monad.IO.Class (liftIO)
import Control.Monad.IO.Control

share [mkPersist sqlSettings, mkMigrate "migrateAll"] [persist|
Person
    name String
    age Int
BlogPost
    title String
    authorId PersonId
|]



step1 :: MonadControlIO m => SqlPersist m ()
step1 = do
  liftIO $ putStrLn "Hello, World"
  runMigration migrateAll
  return ()
  johnId <- insert $ Person "John Doe" 30
  liftIO $ putStrLn "Hello, World"
  janeId <- insert $ Person "Jane Doe" 28
  insert $ BlogPost "My fr1st p0st" johnId
  return ()

-- runSqlConn :: MonadCatchIO m => SqlPersist m a -> Connection -> m a

main :: IO ()
main = withSqliteConn "db.sqlite" $ runSqlConn $ step1

