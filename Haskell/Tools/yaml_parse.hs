{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

import System.IO  
import Text.Printf
import System.Environment (getArgs)
import Data.Object.Yaml
import Data.Object
import Control.Monad
import Control.Failure
import Control.Applicative
import List

getPeopleName :: (Failure ObjectExtractError m)
                 => [(String, Object String String)] -> m String
getPeopleName p = lookupScalar "name" p

peopleNameList :: (Failure ObjectExtractError m)
                  => [[(String, Object String String)]] -> m [String]
peopleNameList people =
    mapM getPeopleName people
--    let a = map getPeopleName people
--    in sequence a
--      return ["aiueo", "kaki"] :: (Failure ObjectExtractError m) => m [String]

fmp :: (Failure ObjectExtractError m) =>
       Object String String -> m [(String, Object String String)]
fmp = fromMapping

fm :: (Failure ObjectExtractError m) =>
      [Object String String] -> [m [(String, Object String String)]]
fm people = map fmp people

convertHTMLListPerson :: (String, [String]) -> String
convertHTMLListPerson (name, cmds) = 
    "<h2>" ++ name ++ "</h2><ul><li>"
    ++ (concat $ intersperse "</li><li>" cmds) ++ "</li></ul>"

convertHTMLList :: [String] -> [[String]] -> String
convertHTMLList names commandsList =
    concat personHTML
        where personHTML = map convertHTMLListPerson $ zip names commandsList

step5 :: (Failure ObjectExtractError m)
         => Object String String -> m String
step5 x = fromScalar x

step4 :: (Failure ObjectExtractError m)
         => [Object String String] -> m [String]
step4 y = mapM step5 y

step3 :: (Failure ObjectExtractError m)
         => [[Object String String]] -> m [[String]]
step3 x = mapM step4 x
--step3 = mapM (mapM fromScalar)

step2 :: (Failure ObjectExtractError m)
         => [(String, Object String String)] -> m [Object String String]
step2 x = lookupSequence "commands" x

step1 :: (Failure ObjectExtractError m)
         =>[[(String, Object String String)]] -> m [[Object String String]]
step1 people = mapM step2 people
--step1 people = mapM (lookupSequence "commands") people


peopleCommandsList :: (Failure ObjectExtractError m)
                  => [[(String, Object String String)]] -> m [[String]]
peopleCommandsList people =
    do
      t <- step1 people 
      u <- step3 t
      return u

processYamlFile :: String -> IO ()
processYamlFile fileName =
    do
      object <- join $ decodeFile fileName :: IO (Object String String)
      people <- fromSequence object
             :: (Failure ObjectExtractError m)
                => m [Object String String]
      let t = fm people
            :: (Failure ObjectExtractError m) =>
               [m [(String, StringObject)]]
      ppl <- sequence t
--            :: (Failure ObjectExtractError m) =>
--               m [[(String, Object String String)]]
      names  <- peopleNameList ppl
             :: (Failure ObjectExtractError m) => m [String]
      commandsList <- peopleCommandsList ppl
                   :: (Failure ObjectExtractError m) => m [[String]]
      let nameInHTML = convertHTMLList names commandsList
      putStrLn $ nameInHTML

main = do
  args <- getArgs
  case args of
    [input] -> processYamlFile input
    _ -> fail "one argument is required"
