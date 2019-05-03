function setMusic(id){
        let player = $("#player");
        player.attr("src", id);
        $('#player').trigger("play");
        let rowid = "#" + id.replace("songs/", "").replace(".mp3", "");
        $( "li" ).each(function( index ) {
          $( this ).css('background-color', 'white');
        });
        let songrow = $(rowid);
        songrow.css('background-color', 'aquamarine');
        $.ajax(
            {
              type : 'POST',
              url : "/queue",
              contentType: "application/json;charset=UTF-8",
              dataType:'json',
              data    : JSON.stringify( { "song_id" : id.replace("songs/", "").replace(".mp3", "") } ),
              success : function(data)
              {
                  console.log(data);
                  $("#queue-songs-list").find("li").remove();
                // alert("alert2:messageServer = "+JSON.stringify(data));
                  for(let row in data) {
                      $("#queue-songs-list").append('<li class="list-group-item">\n' +
                          '<a class="link" onclick = setMusic("' +  data[row][0] + '")>' + data[row][1] + '</a>\n' +
                          '</li>');
                  }
              }
            });
}