{-
$ ghci 3.4_Test.hs
Ok, modules loaded: Test.
Prelude Test> x
5


 $ ghc --make 3.4_Test.hs -o test
[1 of 1] Compiling Main             ( 3.4_Test.hs, 3.4_Test.o )
Linking test ...
 $ ./test 
Hello World!

-}
module Main
       where
x = 5
y = (6, "Hello")
z = x * fst y





signum x =
    if x < 0
    then -1
    else if x > 0
         then 1
         else 0

f x =
    case x of
      0 -> 1
      1 -> 5
      2 -> 2
      _ -> -1

square x = x * x

roots a b c =
    let det = sqrt (b*b - 4*a*c)
    in ((-b + det) / (2 * a),
        (-b - det) / (2 * a))

       
main = putStrLn "Hello World!"