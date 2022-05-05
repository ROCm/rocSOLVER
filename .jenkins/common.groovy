// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()

    String compiler = 'hipcc'
    String hipClang = ''
    String debug = project.buildName.contains('Debug') ? '-g' : ''
    String centos = platform.jenkinsLabel.contains('centos') ? 'source scl_source enable devtoolset-7' : ''
    String noOptimizations = ''

    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        pullRequest.labels.each
        {
            if (it == "noOptimizations")
            {
                noOptimizations = "-n"
            }
        }
    }

    def getRocBLAS = auxiliary.getLibrary('rocBLAS-internal',platform.jenkinsLabel, null, sameOrg)
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getRocBLAS}
                ${auxiliary.exitIfNotSuccess()}
                ${centos}
                ${project.paths.build_command} ${hipClang} ${debug} ${noOptimizations}
                ${auxiliary.exitIfNotSuccess()}
                """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    String buildType = project.buildName.contains('Debug') ? 'debug' : 'release'
    String hmmTestCommand = platform.jenkinsLabel.contains('gfx90a') ? 'HSA_XNACK=1 ./rocsolver-test --gtest_filter=*MANAGED_MALLOC*' : ''

    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}/build/${buildType}/clients/staging
                ./rocsolver-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}
                if [ -f ./test-rocsolver-dlopen ]; then
                  ./test-rocsolver-dlopen --gtest_color=yes
                fi
                ${hmmTestCommand}
                cd ../..
                CTEST_OUTPUT_ON_FAILURE=1 ctest -R '^test-rocsolver-bench'
                """


    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/${buildType}/clients/staging/*.xml"
}

def runPackageCommand(platform, project)
{
        String buildType = project.buildName.contains('Debug') ? 'debug' : 'release'
        def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/${buildType}")
        platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
}

return this


