#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 1 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

import java.nio.file.Path;

rocSOLVERCI:
{

    def rocsolver = new rocProject('rocSOLVER')
    
    def nodes = new dockerNodes(['internal && gfx900 && ubuntu', 'internal && gfx906 && ubuntu'], rocsolver)

    boolean formatCheck = false

    String rocBLAS = auxiliary.getLibrary('rocBLAS','ubuntu', 'develop', true)
    String rocBLAS2 = auxiliary.getLibrary('rocBLAS', 'centos', 'develop', true)

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        rocsolver.paths.build_command = platform.jenkinsLabel.contains('centos') ? 'sudo cmake3 -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hcc ..' :
                                            'sudo cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hcc ..'
        
        def getRocBLAS = platform.jenkinsLabel.contains('centos') ? rocBLAS2 : rocBLAS
        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    ${getRocBLAS}
                    sudo mkdir build && cd build
                    export PATH=/opt/rocm/bin:$PATH
                    ${project.paths.build_command}
                    sudo make -j32
                """

        platform.runCommand(this, command)
    }

    def testType = auxiliary.isJobStartedByTimer() ? '*daily_lapack*' : '*checkin_lapack*'
    def testCommand =
    {
        platform, project->

        try
        {
            def getRocBLAS = platform.jenkinsLabel.contains('centos') ? rocBLAS2 : rocBLAS
            def command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/clients/staging
                        ${getRocBLAS}
                        LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG sudo ./rocsolver-test --gtest_output=xml --gtest_color=yes --gtest_filter=${testType}
                    """

            platform.runCommand(this, command)
        }
        finally
        {
            junit "${project.paths.project_build_prefix}/build/clients/staging/*.xml"
        }        
    }

    def packageCommand =
    {
        platform, project->

        def rpmDeb = platform.jenkinsLabel.contains('centos') ? platform.makePackage('rpm',"${project.paths.project_build_prefix}/build",true) : 
                    platform.makePackage('deb',"${project.paths.project_build_prefix}/build",true)

        packageCommand = platform.jenkinsLabel.contains('hip-clang') ? null : rpmDeb
    }

    buildProject(rocsolver, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}
