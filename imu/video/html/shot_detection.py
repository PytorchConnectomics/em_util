from .html_base import html_base 

class html_shot(html_base):
    def __init__(self, frame_name = '%04d.png', frame_num = 100, frame_start = 0,\
                frame_fps = 30, file_result ='shot_detection.html', num_col = 5):
        self.frame_name = frame_name
        self.frame_num = frame_num
        self.frame_start = frame_start
        self.frame_fps = frame_fps
        self.file_result = file_result
        super().__init__(num_col)

    def getBody(self):
        out = """Shot starting IDs: <textarea id="shot" cols=50 rows=10></textarea> (separated by comma)
        Number of images per row: <textarea id="view" cols=5 rows=1></textarea>
        <br/>
        <button id="sub" style="width:400;height=200">Download CSV</button>
        <div id="img"></div>"""
        return out

    def getHtml(self):
        out = self.getHeader()
        out += self.getBody()
        out += self.getScript()
        return out

    def getScript(self):
        out = '<script src="%s"></script>' % self.file_result
        out += """
        <script>
        var frame_name = "%s";
        var frame_num = %d;
        var frame_start = %d;
        var frame_fps = %d;
        var num_col = %d;
        """ % (self.frame_name, self.frame_num, self.frame_start, self.frame_fps, self.num_col)
        out += """
        var shot_start = [0];
        var shot_selection = [0];
        function loadJs_cb(){
            $('#shot').val(shot_start_str)
            update_value(shot_start_str, shot_selection_str);
        }
        // loadJs(file_result, loadJs_cb)
        loadJs_cb()

        function getImName(i){
            var im_id = frame_start + (i * frame_fps)
            var fn = getFileName(frame_name, im_id);
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
                    lt = frame_num - 1;
                }else{
                    lt = shot_start[i+1] - 1
                }
                out+='<tr><td id="t'+(i)+'" class="shot_sel" style="background-color:'+color_name_shot[shot_selection[i]]+';">'+(i)+"</td><td>"+shot_start[i]+"-"+(lt)+"</td><td>"
                out+='<table>'
                for(j = shot_start[i]; j < lt + 1; j ++){
                    if ((j - shot_start[i]) % num_col == 0){
                        out += '<tr><td>'
                    }
                    out += "" + j
                    out+='<img height=100 src="'+getImName(j)+'">'
                    if ((j - shot_start[i] + 1) % num_col == 0){
                        out +='</td></tr>'
                    }
                }
                if ((lt - shot_start[i] + 1) % num_col != 0){
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
        function update_range(var_col_str){
            num_col = parseInt(var_col_str);
            update_display();
        }
        $("#shot").change(function(){
            var shot_start_str = $(this).val();
            var shot_selection_str = updateArr(shot_start, shot_selection, strToArray(shot_start_str), '0');
            update_value(shot_start_str, shot_selection_str);
        });
        $("#view").change(function () {
            var col_str = $(this).val();
            update_range(col_str)
        });
        $("#sub").click(function(){
            //
            console.log('shot:' + shot_selection);
            alert('This action will download the finalized CSV file!')
            const savedata = []
            for (i = 0; i < shot_start.length; i++) {
                savedata.push("" + shot_start[i] + "," + shot_selection[i] + "," + "%c")
            }
            const blob = new Blob(savedata, { type: 'csv' });
            filename = "%s";
            """ %('\n',self.frame_name[:self.frame_name.rfind("/")]+"_shots.csv")
        out += """
            if (window.navigator.msSaveOrOpenBlob) {
                window.navigator.msSaveBlob(blob, filename);
            }
            else {
                const elem = window.document.createElement('a');
                elem.href = window.URL.createObjectURL(blob);
                elem.download = filename;
                document.body.appendChild(elem);
                elem.click();
                document.body.removeChild(elem);
            }
            // uncomment the code below to save the result on the server
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
        return out

