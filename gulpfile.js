'use strict';
//npm install --save gulp gulp-zip gulp-awslambda
const gulp   = require('gulp');
const zip    = require('gulp-zip');
const path   = require('path');
const lambda = require('gulp-awslambda');
const aws_lamda_tensorflow = require('aws-lambda-tensorflow');

const lambda_params  = {
    FunctionName: 'mylambdafunction', /* Lambda function name */
    Description: 'My tensorflow lambda function that adds two numbers', //Description for your lambda function
    Handler: 'main.handler', //Assuming you will provide main.py file with a function called handler.
    MemorySize: 128,
    Role: 'STRING_VALUE',
    Runtime: 'python2.7',
    Timeout: 10
};

gulp.task('default', () => {
    return gulp.src(['main.py'])
                .pipe(aws_lamda_tensorflow()) //Adds all the required files needed to run tensor flow in aws lambda
                .pipe(zip('archive.zip'))
                .pipe(lambda(lambda_params, opts))
                .pipe(gulp.dest('dist'));
});