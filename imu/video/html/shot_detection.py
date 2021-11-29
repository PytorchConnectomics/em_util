html_shot = """
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://donglaiw.github.io/js/imutil.js"></script>
Shot starting IDs: <textarea id="shot" cols=50 rows=10></textarea> (separated by comma)
<br/>
<button id="sub" style="width:400;height=200">Done</button>
<div id="img"></div>
<form id="mturk_form" method="POST" style="display:none">
     <input id="folder" name="folder" value="">
     <input id="file_id" name="file_id" value="">
     <input id="ans" name="ans">
</form>
<script>
var shot_start = [0];
var shot_selection = [0];
var frame_folder = "%s";
var video_name = "%s";
var genre_name = "./";
var video_url = video_name;
if (video_name.includes('/')){
    genre_name = video_name.substr(0, video_name.lastIndexOf('/'));
    video_url = video_name.substr(video_name.lastIndexOf('/') + 1);
}
var num = %d;
var fps = %d;
function loadJs_cb(){
    $('#shot').val(shot_start_str)
    update_value(shot_start_str, shot_selection_str);
}
loadJs('%s', loadJs_cb)
function getImName(i){
    var im_id = 1 + (i * fps)
    var fn = frame_folder + video_name + "/image_" + printf5d(im_id) + '.png';
    return fn;
}
function update_display(){
    var out=""
    out += "<table border=1>"
    out += '<thead style="display:block;">'
    out += "<tr><td>shot ID</td><td>frame ID</td><td>images</td></tr>"
    out += "</thead>"
    out += '<tbody style="display:block;height:1300px;overflow-y:auto">'
    var lt = 1;
    for(i = 0;i < shot_start.length; i ++){
        if(i == shot_start.length - 1){
            lt = num - 1;
        }else{
            lt = shot_start[i+1] - 1
        }
        out+='<tr><td id="t'+(i)+'" class="shot_sel" style="background-color:'+color_name_shot[shot_selection[i]]+';">'+(i)+"</td><td>"+shot_start[i]+"-"+(lt)+"</td><td>"
        out+='<table>'
        for(j = shot_start[i]; j < lt + 1; j ++){
            if ((j - shot_start[i]) %% numCol == 0){
                out += '<tr><td>'
            }
            out+='<img height=100 src="'+getImName(j)+'">'
            if ((j - shot_start[i] + 1) %% numCol == 0){
                out +='</td></tr>'
            }
        }
        if ((lt - shot_start[i] + 1) %% numCol != 0){
            out += '</td></tr>'
        }
        out += '</table>'
        out += "</td></tr>"
    }
    out += "</tbody>"
    out += "<table>"
    $("#img").html(out)
    $(".shot_sel").click(function() {
        var color_id = getNextColorId($(this)[0].style.backgroundColor, color_name_shot);
        $(this)[0].style.backgroundColor = color_name_shot[color_id];
        var row_id = parseInt($(this)[0].id.substr(1))
        shot_selection[row_id] = color_id;
    });
 
}
function update_value(shot_start_str, shot_selection_str){
    shot_start = strToArray(shot_start_str);
    shot_selection = strToArray(shot_selection_str);
    update_display();
}
$("#shot").change(function(){
    var shot_start_str = $(this).val();
    var shot_selection_str = updateArr(shot_start, shot_selection, strToArray(shot_start_str), '0');
    update_value(shot_start_str, shot_selection_str);
});
$("#sub").click(function(){
    //
    console.log('shot:'+shot_selection);
    /*
    ans_out = $("#shot").val();
    document.getElementById("ans").value = 'var shot_start_str="'+ans_out+'";var shot_selection_str="'+shot_selection+'"';
    document.getElementById("folder").value = get_js_name(false);
    tmp = $.post("../../save_ans.php", $("#mturk_form").serialize(), function(data) {
        window.location=window.location.href.substring(0, window.location.href.lastIndexOf("/"));
    });
    */
  });
</script>
"""
