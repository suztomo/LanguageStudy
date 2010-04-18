ANTLR(ANother Tool for Language Recognition) is a 
parser generator written in Java.

Download antlr-3.2.jar and configure CLASSPATH 
environent value so that it includes 
"antlr-3.2.jar" in current directory.
Then type:
  java org.antlr.Tool
, you will see the error message of ANTLR.
Otherwise, you may find "NoClaDefFoundError: org/antlr/Tool",
in this case, you should check $CLASSPATH and
antlr-3.2.jar in current directory.
