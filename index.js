var through     = require('through2');
var es          = require('event-stream');
var vinyl       = require('vinyl-fs');
var path        = require('path');
const dependencyDirectory = path.resolve(__dirname,'dependencies');

module.exports =  function () {
    var pass = through.obj();
    return es.duplex(pass, es.merge(vinyl.src.apply(vinyl.src, [dependencyDirectory+'\\*']), pass));
};