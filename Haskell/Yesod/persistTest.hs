{-# LANGUAGE QuasiQuotes, TemplateHaskell, TypeFamilies, OverloadedStrings #-}
{-# LANGUAGE GADTs #-}
import Database.Persist
import Database.Persist.Sqlite
import Database.Persist.TH
import Control.Monad.IO.Class (liftIO)
    
share [mkPersist sqlSettings, mkMigrate "migrateAll"] [persist|
Person
    name String
    age Int
BlogPost
    title String
    authorId PersonId
|]

main :: IO ()
main = withSqliteConn "db.sqlite" $ runSqlConn $ do
    runMigration migrateAll
    lst <- sequence [ insert $ Person "John Doe" $ x * 4 | x<-[1..6] ]
    let johnId = lst !! 0
    janeId <- insert $ Person "Jane Doe" 28
    insert $ BlogPost "My fr1st p0st" johnId
    insert $ BlogPost "One more for good measure" johnId
    oneJohnPost <- selectList [BlogPostAuthorId ==. johnId] [LimitTo 3]
    liftIO $ print (oneJohnPost :: [(BlogPostId, BlogPost)])

    john <- get johnId
    liftIO $ print (john :: Maybe Person)

    people <- selectList [PersonAge >. 25, PersonAge <=. 30] []
    delete janeId¯
    deleteWhere [BlogPostAuthorId ==. johnId]
