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

String getRocBLAS = """ 
        sudo wget http://10.216.151.18:8080/job/ROCmSoftwarePlatform/job/rocBLAS/job/develop/lastSuccessfulBuild/artifact/*zip*/archive.zip
                sudo unzip archive.zip
                sudo dpkg -i archive/*/*/*/*/*/*.deb
    """

rocSOLVERCI:
{

    def rocsolver = new rocProject('rocSOLVER')
    // customize for project
    rocsolver.paths.build_command = 'sudo cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hcc ..'

    // Define test architectures, optional rocm version argument is available
    def firstNode = new dockerNodes(['gfx906 && ubuntu && rs'], rocsolver)
    def secondNode = new dockerNodes(['gfx900 && ubuntu && rs'], rocsolver)

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

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

    def testCommand =
    {
        platform, project->

        def command

        if(auxiliary.isJobStartedByTimer())
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/clients/staging
		            ${getRocBLAS}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG sudo ./rocsolver-test --gtest_output=xml --gtest_color=yes --gtest_filter=*daily_lapack*
                """

            platform.runCommand(this, command)
            junit "${project.paths.project_build_prefix}/build/clients/staging/*.xml"
        }
        else
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/clients/staging
		            ${getRocBLAS}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG sudo ./rocsolver-test --gtest_output=xml --gtest_color=yes  --gtest_filter=*checkin_lapack*
                """
            
            platform.runCommand(this, command)
            junit "${project.paths.project_build_prefix}/build/clients/staging/*.xml"
        }
    }

    def packageCommand =
    {
        platform, project->

        def command

        if(platform.jenkinsLabel.contains('hip-clang'))
        {
            packageCommand = null
        }
        else if(platform.jenkinsLabel.contains('centos'))
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build
                    sudo make package
                    sudo mkdir -p package
                    sudo mv *.rpm package/
                    sudo rpm -qlp package/*.rpm
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/package/*.rpm""")
        }
        else
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    sudo make package
                    sudo mkdir -p package
                    sudo mv *.deb package/
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/package/*.deb""")
        }
    }

    buildProject(rocsolver, formatCheck, firstNode.dockerArray, compileCommand, testCommand, packageCommand)
    buildProject(rocsolver, formatCheck, secondNode.dockerArray, compileCommand, testCommand, packageCommand)

}
