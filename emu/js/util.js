/**
 * Get the value of a URL parameter.
 *
 * @param {string} name - The name of the parameter.
 * @returns {string} - The value of the parameter, or an empty string if not found.
 */
function get_url_param(name) {
    var rx = new RegExp('[\&|\?]' + name + '=([^\&\#]+)'),
        val = window.location.search.match(rx);
    return val ? val[1] : '';
}

/**
 * Get the unique elements from an array.
 *
 * @param {*} value - The current value being processed.
 * @param {number} index - The index of the current value.
 * @param {Array} self - The array being processed.
 * @returns {boolean} - Whether the value is unique in the array.
 */
function get_unique(value, index, self) {
    return self.indexOf(value) === index;
}
