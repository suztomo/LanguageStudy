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
    val ircWriter:IrcWriter = ircClient.getWriter
    val httpMessageClient = new HttpMessageClient(config)
    config.mailToName = configReader.read("nickname.json").asInstanceOf[Map[String,String]]
    def ircMsgHandler(sender:String, channelName:String, text:String) {
      val msg = sender + ":" + text
      println(msg)
//      xmppWriter ! XmppBroadcastMsg(sender + ": " + text)
      httpMessageClient ! XmppBroadcastMsg(msg)
    }
    def xmppMsgHandler(sender:String, text:String, self:XmppClient) {
      val nick = try {
        config.mailToName(sender)
      } catch {
        case e:java.util.NoSuchElementException => {
          sender.split("@", 2)(0)
        }
      }
      var messageLine:String = null
      if (sender == "<config>") {
        ircWriter ! Notice(config.ircRoom, text)
      } else {
        messageLine = nick + ":" + text
        ircWriter ! PrivMsg(config.ircRoom, messageLine)
      }
      println(messageLine)
      httpMessageClient ! XmppBroadcastMsgFrom(messageLine, sender)
      for (c <- self.chats) {
        if (c != self.chat) {
          xmppWriter ! XmppMsg(c, messageLine)
        }
      }
    }
    ircClient.setMsgHandler(ircMsgHandler _)
    xmppClient.setMsgHandler(xmppMsgHandler _)
    ircClient.start
    xmppClient.start
    httpMessageClient.start
//    Thread.sleep(10000)
  }
}
