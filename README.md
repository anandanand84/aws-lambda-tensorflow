# aws-lambda-tensorflow

A gulp plugin to deploy python tensorflow in aws lambda. 

## Prerequisites

This plugin assumes you have aws-cli installed and configured with proper access rights to use aws lambda.

## Installation

```
npm install --save-dev aws-lambda-tensorflow
```

## Usage

```javascript

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
    MemorySize: 512,
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

```

## API

### aws_lamda_tensorflow(options)

#### options

##### region(mandatory)

Six lambda available regions.

Type: `string`<br>
values: `ap-northeast-1, ap-southeast-2, eu-central-1, eu-west-1, us-east-1, us-west-2`

##### usePluginDependency (either one of usePluginDependency, virtualEnvPath, dependencies is required)

Use pre built dependencies generated on specific region. This is the preferred method.

Type: `boolean`<br>
Default: `false`

##### virtualEnvPath (either one of usePluginDependency, virtualEnvPath, dependencies is required)

If virtualEnv is used during development, provide path of the directory. Adding virtual evn libraries from local machine will cause it to compile with incorrect architecture since Aws lambda may use a different architecture than local machine. If you have correct architecutre prefer dependency path options than virtualEnv

Type: `string`<br>
Example: `~/tensorflow`

##### dependencies (either one of usePluginDependency, virtualEnvPath, dependencies is required)

List of globbing patterns similar to gulp.src to add additional dependencies.

Type: `Array`<br>


## Sample Project

[sample-tensorflow-aws-lambda](https://github.com/anandanand84/sample-aws-lambda-tensorflow)

## Contributing

Just give a pull request.

See Creating-Dependency.md. 