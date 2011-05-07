package net.suztomo.ponta

class EmptyIrcHost extends Throwable

@serializable
class Options(var ircServer:String = "cs2009.org",
              var ircPort:Int = 6666,
              var xmppServer:String = "talk.google.com",
              var xmppPort:Int = 5222,
              var xmppBot:String = "suztomobot@gmail.com",
              var xmppPass:String = "takokome",
              var ircNick:String = "ponta",
              var ircSsl:Boolean = true,
              var ircRoom:String = "#cs2009",
              var ignoreCertificate:Boolean = true,
              var mailToName:Map[String,String] = null,
              var xmppRecipients:Array[String] = Array("tomotomotomo888@gmail.com"),
              var httpMessageURL:String = "http://cs2009irc.appspot.com/message"
            )
{

}
