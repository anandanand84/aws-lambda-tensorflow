'use strict';
//npm install --save gulp gulp-zip gulp-awslambda
const gulp   = require('gulp');
const zip    = require('gulp-zip');
const path   = require('path');
const lambda = require('gulp-awslambda');
const aws_lamda_tensorflow = require('aws-lambda-tensorflow');

const lambda_params  = {
    FunctionName: 'addtensorflow', /* Lambda function name */
    Description: 'My tensorflow lambda function that adds two numbers', //Description for your lambda function
    Handler: 'simple_add.lambda_handler', //Assuming you will provide main.py file with a function called handler.
    MemorySize: 128,
    Runtime: 'python2.7',
    Role : 'ROLE_STRING',//eg:'arn:aws:iam::[Account]:role/lambda_basic_execution'
    Timeout: 50
};

var opts = {
    region : 'ap-southeast-2'
}

gulp.task('default', () => {
    return gulp.src(['simple_add.py'])
        .pipe(aws_lamda_tensorflow({ region : 'ap-southeast-2', usePluginDependency  : true })) //Adds all the required files needed to run tensor flow in aws lambda
        .pipe(zip('archive.zip'))
        .pipe(lambda(lambda_params, opts))
        .pipe(gulp.dest('dist'));
});