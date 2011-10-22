import Text.ParserCombinators.Parsec
import Text.ParserCombinators.Parsec.Expr
import Text.ParserCombinators.Parsec.Language
import Debug.Trace
import qualified Text.ParserCombinators.Parsec.Token as P
import Data.Char

data Identifier = Identifier String deriving (Show, Read, Eq)

data FuncDef = FuncDef {
      name :: Identifier,
      fileName :: String,
      lineNum :: Int,
      body :: Block } deriving Show

perlFile :: GenParser Char st [ FuncDef ]
perlFile =
    do result <- many funcDef
       eof
       return result

languageDefinition = javaStyle {
                       reservedNames = ["sub", "if", "$"]
                     , reservedOpNames = ["*", "/", "+", "-", "="]
                     }

lexer = P.makeTokenParser languageDefinition

reserved              = P.reserved lexer
symbol                = P.symbol lexer
identifierLexer       = P.identifier lexer  
whiteSpace            = P.whiteSpace lexer
reservedOp            = P.reservedOp lexer
lexeme                = P.lexeme lexer

identifier = do {
  x <- identifierLexer;
  return(Identifier x)
}

funcDef :: GenParser Char st FuncDef
funcDef =
    do
      reserved "sub"
      pos <- getPosition
      name <- identifierLexer
      symbol "{"
      body <- funcBody
      symbol "}"
      return (FuncDef (Identifier name) (sourceName pos) (sourceLine pos) body)

funcBody :: GenParser Char st Block
funcBody = do
  block

data Variable = Variable String
              deriving Show

variable :: GenParser Char st Variable
variable = lexeme $ do
  char '$';
  Identifier name <- identifier
  return (Variable name)

data Expression = VariableExp String
         | ConstantExp String
         | NoneExp
    deriving Show

expression :: GenParser Char st Expression
expression = lexeme $ try (do{
  Variable name <- variable;
  return (VariableExp name)
}) <|> try (do{
  d <- many1 digit;
  return (ConstantExp d)
})

data Statement = AssignStatement Expression Expression
               | IfStatement Expression Expression Expression
               | NoneStatement
    deriving Show
type Block = [ Statement ]
--               | FuncCallStatement Expression [Expression]

block :: GenParser Char st [ Statement ]
block = do
  lexeme $ many statement

statement :: GenParser Char st Statement
statement = try (do {
  lvalue <- expression;
  reservedOp "=";
  rvalue <- expression;
  return (AssignStatement lvalue rvalue)
}) <|> try (do {
  reserved "if";
  cond <- expression;
  expression1 <- expression;
  expression2 <- expression;
  return (IfStatement cond expression1 expression2)
})



parsePerlFile :: String -> Either ParseError [ FuncDef ]
parsePerlFile input = parse perlFile "SampleFileName" input

t :: Either ParseError [ FuncDef ]
--t = parsePerlFile "sub tomeFunc { if 12 $a1 3 }"

t = parsePerlFile "sub tomeFunc { if 12 3 4 } \n\n sub takaFunc { if 9 21 23 } "
