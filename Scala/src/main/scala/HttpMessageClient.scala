package net.suztomo.ponta

import scala.collection.JavaConversions._
import org.jivesoftware.smack._
import scala.actors._, Actor._
import java.util.ArrayList

import org.apache.http.protocol._
import org.apache.http._
import org.apache.http.message._
import org.apache.http.client.methods.HttpPost
import org.apache.http.client.entity._
import org.apache.http.entity.StringEntity
import org.apache.http.auth.{AuthScope,UsernamePasswordCredentials}
import org.apache.http.impl.client.DefaultHttpClient
import org.apache.http.params.CoreProtocolPNames

import java.io.{BufferedReader,InputStreamReader}

class HttpMessageClient(var config:Options) extends Actor {
  val client = new DefaultHttpClient
  def sendPost(message:String, sender:String = null) {
    val post = new HttpPost(config.httpMessageURL)
    post.addHeader("Content-Type", "application/x-www-form-urlencoded")
    var parameters:ArrayList[NameValuePair] = new ArrayList[NameValuePair];
    parameters.add(new BasicNameValuePair("text", message))
    if (sender != null) {
      parameters.add(new BasicNameValuePair("sender", sender))
    }
    val ent:UrlEncodedFormEntity = new UrlEncodedFormEntity(parameters,
                                                            HTTP.UTF_8);
    post.setEntity(ent)
    val response = client.execute(post)
    val statusLine = response.getStatusLine()
    println(statusLine.getReasonPhrase() + statusLine.getStatusCode() )
    val input = new BufferedReader(new InputStreamReader(response.getEntity.getContent))
    var line = ""
    println("posted to " + config.httpMessageURL)
    while({line=input.readLine; line != null}) {
      println("HTTP: " + line)
    }
  }
  def act() {
    loop{
      react{
        case XmppBroadcastMsg(message) => {
          sendPost(message)
        }
        case XmppBroadcastMsgFrom(message, sender) => {
          sendPost(message, sender)
        }
      }
    }
  }

}
