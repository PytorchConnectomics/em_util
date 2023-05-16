function get_url_param(name) {
    var rx = new RegExp('[\&|\?]' + name + '=([^\&\#]+)'),
        val = window.location.search.match(rx);
    return !val ? '' : val[1];
}
function get_unique(value, index, self) {
    return self.indexOf(value) === index;
}
