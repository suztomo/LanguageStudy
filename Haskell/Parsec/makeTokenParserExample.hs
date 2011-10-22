import Text.ParserCombinators.Parsec
import Text.ParserCombinators.Parsec.Expr
import qualified Text.ParserCombinators.Parsec.Token as P
import Text.ParserCombinators.Parsec.Language

data Expr = Var String | Con Bool | Uno Unop Expr | Duo Duop Expr Expr
    deriving Show
data Unop = Not deriving Show
data Duop = And | Iff deriving Show
data Stmt = Nop | String := Expr | If Expr Stmt Stmt | While Expr Stmt
          | Seq [Stmt]
    deriving Show

lexer :: P.TokenParser ()
lexer  = P.makeTokenParser 
         (haskellDef
         { reservedOpNames = ["*","/","+","-"]
         })

whiteSpace= P.whiteSpace lexer
natural   = P.natural lexer
parens    = P.parens lexer
reservedOp= P.reservedOp lexer


factor = parens expr
        <|> natural
        <?> "simple expression"
table = [[op "*" (*) AssocLeft, op "/" div AssocLeft]
        ,[op "+" (+) AssocLeft, op "-" (-) AssocLeft]
        ]
      where
        op s f assoc
          = Infix (do{ reservedOp s; return f}) assoc
expr :: Parser Integer
expr = buildExpressionParser table factor
      <?> "expression"

runLex :: Show a => Parser a -> String -> IO ()
runLex p input = parseTest (do{ whiteSpace
                    ; x <- p
                    ; eof
                    ; return x
                    }) input
t :: IO ()
t = do
  runLex expr "1+(2*3)"
  
