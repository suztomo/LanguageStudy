import Text.ParserCombinators.Parsec
import Text.ParserCombinators.Parsec.Expr
import Text.ParserCombinators.Parsec.Token
import Text.ParserCombinators.Parsec.Language

data FuncDef = FuncDef {
      name :: String,
      body :: String } deriving Show

perlFile :: GenParser Char st [ FuncDef ]
perlFile =
    do result <- many funcDef
       eof
       return result

languageDefinition = javaStyle {
                       reservedNames = ["sub"]
                     }


funcDef :: GenParser Char st FuncDef
funcDef =
    do
      char 's'
      char 'u'
      char 'b'
      char ' '
      name <- funcName 
      char ' '
      char '{'
      body <- funcBody
      char '}'
      return (FuncDef name body)

funcName :: GenParser Char st String
funcName =
    do many (noneOf "\n{} ")

funcBody :: GenParser Char st String
funcBody =
    do many (noneOf "\n{}")

parsePerlFile :: String -> Either ParseError [ FuncDef ]
parsePerlFile input = parse perlFile "(unknown)" input

t :: Either ParseError [ FuncDef ]
t = parsePerlFile "sub tomeFunc { aiueo }"
