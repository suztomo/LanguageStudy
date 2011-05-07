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
    var mailToIrcClient: Map[String,IrcClient] = Map()
    def updateMailToName() {
      config.mailToName = configReader.read("nickname.json").asInstanceOf[Map[String,String]]
    }
    def ircMsgHandler(nickname:String, channelName:String, text:String) {
      updateMailToName()
      val msg = nickname + ":" + text
      var senderEmail:Option[String] = None
      config.mailToName foreach(
        (t) => {
          if (nickname.indexOf(t._2) != -1) {
            senderEmail = Some(t._1)
          }
        }
      )
      senderEmail match {
        case Some(email) => {
          httpMessageClient ! XmppBroadcastMsgFrom(msg, email)
        }
        case None => {
          httpMessageClient ! XmppBroadcastMsg(msg)
        }
      }
    }
    def getNick(email:String):String = {
      updateMailToName()
      val nick = try {
        config.mailToName(email)
      } catch {
        case e:java.util.NoSuchElementException => {
          email.split("@", 2)(0)
        }
      }
      nick
    }
    def xmppMsgHandler(sender:String, text:String, self:XmppClient) {
      var messageLine:String = null
      sender match {
        case "<config>" => {
//          ircWriter ! Notice(config.ircRoom, text)
        }
        case "<join>" => {
          val senderMail = text
          mailToIrcClient.get(senderMail) match {
            case Some(_) => {
              println("ingoring existing client")
            }
            case None => {
              val n = getNick(senderMail)
              val o = new Options(ircNick = n)
              val ic = new IrcClient(o)
              println("created ircClient for " + senderMail)
              mailToIrcClient += senderMail -> ic
              ic.start
            }
          }
        }
        case "<part>" => {
          val senderMail = text
          mailToIrcClient.get(senderMail) match {
            case Some(ic) => {
              val iw = ic.getWriter
              iw ! Part(config.ircRoom)
              iw ! Disconnect()
              (ic.getPinger) ! Stop()
              mailToIrcClient -= senderMail
            }
            case None => {
              println("Invalid email when part: " + senderMail)
            }
          }
        }
        case _ => {
          val nick = getNick(sender)
          mailToIrcClient.get(sender) match {
            case Some(ic) => {
              val iw = ic.getWriter
              iw ! PrivMsg(config.ircRoom, text)
            }
            case None => {
              val o = new Options(ircNick = nick)
              val ic = new IrcClient(o)
              println("created (existing) ircClient for " + sender)
              mailToIrcClient += sender -> ic
              ic.start
              val iw = ic.getWriter
              Thread.sleep(1000)
              iw ! PrivMsg(config.ircRoom, text)
            }
          }
              /*
               * Send messsage to all registered users via Google App Engine
               */

/*
        for (c <- self.chats) {
          if (c != self.chat) {
            xmppWriter ! XmppMsg(c, messageLine)
          }
        }
*/
        }
      }
    }
    updateMailToName()
    ircClient.setMsgHandler(ircMsgHandler _)
    xmppClient.setMsgHandler(xmppMsgHandler _)
    ircClient.start
    xmppClient.start
    httpMessageClient.start
//    Thread.sleep(10000)
  }
}
