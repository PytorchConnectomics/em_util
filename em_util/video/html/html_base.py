class html_base(object):
    def __init__(self, num_col=7):
        self.num_col = num_col

    def getHeader(self):
        out = """<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <script src="https://donglaiw.github.io/js/imutil.js"></script>
        """
        return out

    def getForm(self):
        out = """<form id="mturk_form" method="POST" style="display:none">
             <input id="folder" name="folder" value="">
             <input id="file_id" name="file_id" value="">
             <input id="ans" name="ans">
        </form>
        """
        return out

    def getBody(self):
        raise NotImplementedError()

    def getScript(self):
        raise NotImplementedError()

    def getHtml(self):
        raise NotImplementedError()
