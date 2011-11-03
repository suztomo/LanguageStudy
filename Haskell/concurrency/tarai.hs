tarai :: Int -> Int -> Int -> Int
tarai x y z
    | x <= y    = z
    | otherwise = tarai(tarai (x-1) y z)
                       (tarai (y-1) z x)
                       (tarai (z-1) x y)

main :: IO ()
main = putStrLn . show $ tarai  31 22 11
