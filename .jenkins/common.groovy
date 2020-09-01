// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()

    String compiler = jobName.contains('hipclang') ? 'hipcc' : 'hcc'
    String hipClang = jobName.contains('hipclang') ? '--hip-clang' : ''
    String sles = platform.jenkinsLabel.contains('sles') ? '/usr/bin/sudo --preserve-env' : ''
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

    def getRocBLAS = auxiliary.getLibrary('rocBLAS-internal',platform.jenkinsLabel,'develop', sameOrg)
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getRocBLAS}
                ${auxiliary.exitIfNotSuccess()}
                ${centos}
                ${sles} ${project.paths.build_command} ${hipClang} ${debug} ${noOptimizations}
                ${auxiliary.exitIfNotSuccess()}
                """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release/clients/staging
                ${sudo} GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsolver-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}
                """

    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
}

def runPackageCommand(platform, project)
{
        String buildType = project.buildName.contains('Debug') ? 'debug' : 'release'
        def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/${buildType}")
        platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
}

return this


