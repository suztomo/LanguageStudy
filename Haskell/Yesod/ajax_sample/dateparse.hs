import Yesod


main = do
    let m = parseDate "2009/03/03"
    case m of
      Left msg -> putStrLn $ "err"
      Right d -> putStrLn $ show d
