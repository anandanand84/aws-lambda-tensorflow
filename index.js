var through     = require('through2');
var vinyl       = require('vinyl-fs');
var gutil       = require('gulp-util');
var PluginError = gutil.PluginError;
var path        = require('path');
var gulp        = require('gulp');
const fs        = require('fs');
const unzip     = require('gulp-unzip');
const GITHUB_URL = 'https://cdn.rawgit.com/anandanand84/aws-lambda-tensorflow-dependencies/master/';
const dependencyDirectory = path.resolve(__dirname, 'dependencies');
var https         = require('https');

module.exports =  function (opts) {
    opts.version = opts.version || 'LATEST';

    if(!opts.region) {
        console.error(new PluginError('aws-lambda-tensorflow',  'region is mandatory'));
        return;
    }

    var addTensorDependencyPath = function(originalStream, endCallback) {
        gulp.src(path.resolve(__dirname, 'dependencies'+path.sep+'lambda-tensorflow-dependency-'+opts.region+'-'+opts.version)+path.sep+'**'+path.sep+'*', { dot : true})
        .pipe(through.obj(function(file, encoding, cb){
            originalStream.push(file);
            cb();
        }))
        .on('finish', function() {
            console.log('finish')
            endCallback();
        })
    };

    var pass = through.obj(function(file, encoding, callback) {
        callback(null, file);
    },function(callback) {
        const requiredDependency = path.resolve(dependencyDirectory, 'lambda-tensorflow-dependency-'+opts.region+'-'+opts.version);
        var self = this;
        if((!fs.existsSync(requiredDependency))){
            var fileStream = fs.createWriteStream(dependencyDirectory+path.sep+'lambda-tensorflow-dependency-'+opts.region+'-'+opts.version+'.zip');
            var remoteFileLocation = GITHUB_URL+opts.version+'/'+'lambda-tensorflow-dependency-'+opts.region+'.zip';
            console.log('Dependency not available, Downloading dependency file '+remoteFileLocation);
            https.get(remoteFileLocation, function(response) {
                response.pipe(fileStream)
                fileStream.on('finish', function() {
                    console.log('Download Complete');
                    gulp.src(dependencyDirectory+path.sep+'lambda-tensorflow-dependency-'+opts.region+'-'+opts.version+'.zip' , { dot : true })
                    .pipe(unzip({ keepEmpty : true }))
                    .pipe(gulp.dest(requiredDependency))
                    .on('finish', function() {
                        addTensorDependencyPath(self, callback);
                    })
                })
            });
        }else {
            addTensorDependencyPath(self, callback);
        }
    });

    return pass;
};