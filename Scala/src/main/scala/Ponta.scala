package net.suztomo.ponta

import scopt._

object Ponta {
  val VERSION = "0.1"
  def main(args: Array[String]) {
    val config = new Options()
    val parser = new OptionParser {
      opt("s", "server", "server name", {v: String => config.ircServer = v})
      opt("t", "xserver", "xmpp server name", {v:String => config.xmppServer = v})
      intOpt("p", "port", "server port", {p: Int => config.ircPort = p})
      intOpt("q", "xport", "xmpp server port", {p: Int => config.xmppPort = p})
      booleanOpt("S", "SSL", "use SSL for IRC", {b: Boolean => config.ircSsl = b})
/*      arg("whatnot", "some argument", {v: String => config.whatnot = v}) */
    }
    if (! parser.parse(args)) {
      println("Invalid option")
      exit(1)
    }
    /*
     Any abstraction over two kinds of communication ?
     */
    val xmppClient = new XmppClient(config)
    val xmppWriter:XmppWriter = xmppClient.getWriter
    val ircClient = new IrcClient(config)
    def ircMsgHandler(sender:String, channelName:String, text:String) {
      println(sender + ": " + text)
      xmppWriter ! Msg(sender + ": " + text)
    }
    def xmppMsgHandler(sender:String, text:String) {
      println("Hi, xmppMsgHandler! " + text)
    }
    ircClient.setMsgHandler(ircMsgHandler _)
    xmppClient.setMsgHandler(xmppMsgHandler _)
    ircClient.start
    xmppClient.start
    Thread.sleep(10000)
  }
}
