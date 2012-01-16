function processUserInfo(obj) {
    console.log(obj);
    $('#user_info').text(obj.name);
    $('#user_info').append($('<small>&nbsp;<a href="/logout">Logout</a></small>'));
    $('#user_info').style("font-size: 1.2em");
}

$(document).ready(function() {
    var url = "/user/info";
    $.ajax({
        url : url,
        dataType: "json",
        success: processUserInfo
    });
});
