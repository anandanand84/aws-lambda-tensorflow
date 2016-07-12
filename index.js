var through     = require('through2');
var es          = require('event-stream');
var vinyl       = require('vinyl-fs');
var gutil       = require('gulp-util');
var PluginError = gutil.PluginError;
var path        = require('path');
const dependencyDirectory = path.resolve(__dirname);
const fs        = require('fs');

module.exports =  function (opts) {
    var pass = through.obj();
    var dependencyPath = [];
    if(opts.virtualEnvPath && (typeof opts.virtualEnvPath === 'string')) {
        var virtualEnvLibraries = path.normalize(opts.virtualEnvPath + path.sep + 'lib' + path.sep + 'python2.7'+path.sep+'site-packages');
        stats = fs.lstatSync(virtualEnvLibraries)
        if(stats.isDirectory()) {
            console.log('Adding virtual evn libraries from local machine will cause it to compile with incorrect architecture since Aws lambda may use a different architecture than local machine. ' +
                'If you have correct architecutre use dependency path options than virtualEnv');
            dependencyPath.push(virtualEnvLibraries+path.sep+'**'+path.sep+'*');
        } else {
            console.error('Could not find directory '+virtualEnvLibraries)
            this.emit('error', new PluginError('aws-lambda-tensorflow',  'Could not find directory '+virtualEnvLibraries));
            return;
        }
    } else if (opts.usePluginDependency){
        console.log('virtualEnvPath Not provided or virtualEnvPath not a string, using the plugin inbuilt dependencies')
        dependencyPath.push(path.resolve(__dirname, 'lambda-tensorflow-dependency-'+opts.region)+path.sep+'**'+path.sep+'*')
    }
    else if(opts.dependencies && opts.dependencies instanceof Array) {
        dependencyPath = dependencyPath.concat(opts.dependency)
    }else {
        console.error('Provide either of usePluginDependency, dependencies or virtualEnvPath' );
        this.emit('error', new PluginError('aws-lambda-tensorflow',  'Provide either of usePluginDependency, dependencies or virtualEnvPath'));
    }
    console.log('Using dependency path',dependencyPath)
    return es.duplex(pass, es.merge(vinyl.src.apply(vinyl.src, [dependencyPath, { 'dot' : true}]), pass));
};