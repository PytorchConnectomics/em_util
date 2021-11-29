function getUrlParam(name){
    var rx = new RegExp('[\&|\?]'+name+'=([^\&\#]+)'),
    val = window.location.search.match(rx);
    return !val ? '':val[1];
}
function getUnique(value, index, self) {
    return self.indexOf(value) === index;
}
